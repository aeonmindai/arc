//! Capture-once CUDA graph with private memory pool for isolation.
//!
//! The private pool ensures graph-owned memory doesn't interfere with
//! the rest of the application (PagedAttention KV cache, model weights, etc.).
//!
//! Sequence:
//! 1. Save device's current memory pool
//! 2. Create private pool, set RELEASE_THRESHOLD=UINT64_MAX (never release)
//! 3. Install private pool as device default
//! 4. cuStreamBeginCapture → forward pass → cuStreamEndCapture
//! 5. Restore original pool
//! 6. cuGraphInstantiate
//! 7. On replay: cuGraphLaunch (graph uses its own pool addresses)

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
    /// Private memory pool for this graph's allocations
    pool: CUmemoryPool,
}

#[cfg(feature = "cuda")]
impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            cuGraphExecDestroy(self.exec);
            cuMemPoolDestroy(self.pool);
        }
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    stream: CUstream,
    device_ordinal: CUdevice,
    graphs: HashMap<usize, CapturedGraph>,
    enabled: bool,
    warmup_remaining: u32,
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
            tracing::warn!("CUDA graph: NULL stream, capture disabled");
            return Ok(Self {
                stream, device_ordinal: 0,
                graphs: HashMap::new(), enabled: false, warmup_remaining: 0,
            });
        }

        // Get device ordinal from the stream's context
        let ordinal = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as CUdevice,
            _ => 0,
        };

        tracing::info!("CUDA graph: non-null stream on device {ordinal}, capture enabled");

        Ok(Self {
            stream, device_ordinal: ordinal,
            graphs: HashMap::new(), enabled: true, warmup_remaining: warmup_steps,
        })
    }

    pub fn is_enabled(&self) -> bool { self.enabled && self.warmup_remaining == 0 }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("CUDA graph: warmup done");
            }
            true
        } else { false }
    }

    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Replay a previously captured graph. Returns the output tensor.
    pub fn replay(&self, batch_size: usize) -> candle_core::Result<Tensor> {
        let captured = self.graphs.get(&batch_size).ok_or_else(|| {
            candle_core::Error::Msg(format!("No graph for batch_size={batch_size}"))
        })?;
        unsafe {
            let s = cuGraphLaunch(captured.exec, self.stream);
            if s != CUDA_SUCCESS { candle_core::bail!("cuGraphLaunch failed: {s}"); }
            cudaStreamSynchronize(self.stream);
        }
        Ok(captured.output.clone())
    }

    /// Create a private memory pool for graph capture.
    fn create_private_pool(&self) -> candle_core::Result<CUmemoryPool> {
        let props = CUmemPoolProps {
            alloc_type: CUmemAllocationType::PINNED,
            handle_type: CUmemHandleType::NONE,
            location: CUmemLocation {
                loc_type: CUmemLocationType::DEVICE,
                id: self.device_ordinal,
            },
            win32_security_attributes: std::ptr::null_mut(),
            max_size: 0,
            usage: 0,
            reserved: [0u8; 54],
        };

        let mut pool: CUmemoryPool = std::ptr::null_mut();
        let s = unsafe { cuMemPoolCreate(&mut pool, &props) };
        if s != CUDA_SUCCESS {
            candle_core::bail!("cuMemPoolCreate failed: {s}");
        }

        // Never release memory back to OS — keep addresses stable for replay
        let mut threshold: u64 = u64::MAX;
        let s = unsafe {
            cuMemPoolSetAttribute(
                pool,
                CUmempoolAttribute::RELEASE_THRESHOLD,
                &mut threshold as *mut u64 as *mut _,
            )
        };
        if s != CUDA_SUCCESS {
            unsafe { cuMemPoolDestroy(pool); }
            candle_core::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {s}");
        }

        Ok(pool)
    }

    /// Begin graph capture with a private memory pool.
    /// Returns the pool and saved original pool for restoration.
    pub fn begin_capture(&mut self, batch_size: usize) -> candle_core::Result<(CUmemoryPool, CUmemoryPool)> {
        // Create private pool
        let graph_pool = self.create_private_pool()?;

        // Save original pool
        let mut original_pool: CUmemoryPool = std::ptr::null_mut();
        let s = unsafe { cuDeviceGetMemPool(&mut original_pool, self.device_ordinal) };
        if s != CUDA_SUCCESS {
            unsafe { cuMemPoolDestroy(graph_pool); }
            candle_core::bail!("cuDeviceGetMemPool failed: {s}");
        }

        // Install private pool
        let s = unsafe { cuDeviceSetMemPool(self.device_ordinal, graph_pool) };
        if s != CUDA_SUCCESS {
            unsafe { cuMemPoolDestroy(graph_pool); }
            candle_core::bail!("cuDeviceSetMemPool (install) failed: {s}");
        }

        // Sync stream before capture
        unsafe { cudaStreamSynchronize(self.stream); }

        // Begin capture
        let s = unsafe {
            cuStreamBeginCapture_v2(self.stream, CUstreamCaptureMode::THREAD_LOCAL)
        };
        if s != CUDA_SUCCESS {
            // Restore original pool before bailing
            unsafe { cuDeviceSetMemPool(self.device_ordinal, original_pool); }
            unsafe { cuMemPoolDestroy(graph_pool); }
            self.enabled = false;
            candle_core::bail!("cuStreamBeginCapture failed: {s}");
        }

        tracing::info!("CUDA graph: capture started for batch_size={batch_size} with private pool");
        Ok((graph_pool, original_pool))
    }

    /// End capture, restore original pool, instantiate graph, cache it.
    pub fn end_capture_and_cache(
        &mut self,
        batch_size: usize,
        output: Tensor,
        graph_pool: CUmemoryPool,
        original_pool: CUmemoryPool,
    ) -> candle_core::Result<Tensor> {
        // End capture
        let mut graph: CUgraph = std::ptr::null_mut();
        let s = unsafe { cuStreamEndCapture(self.stream, &mut graph) };

        // ALWAYS restore original pool
        unsafe { cuDeviceSetMemPool(self.device_ordinal, original_pool); }

        if s != CUDA_SUCCESS {
            unsafe { cuMemPoolDestroy(graph_pool); }
            self.enabled = false;
            candle_core::bail!("cuStreamEndCapture failed: {s}");
        }

        // Instantiate
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let s = unsafe {
            cuGraphInstantiate_v2(
                &mut exec, graph,
                std::ptr::null_mut(), std::ptr::null_mut(), 0,
            )
        };
        unsafe { cuGraphDestroy(graph); }
        if s != CUDA_SUCCESS {
            unsafe { cuMemPoolDestroy(graph_pool); }
            self.enabled = false;
            candle_core::bail!("cuGraphInstantiate failed: {s}");
        }

        // First launch
        let s = unsafe { cuGraphLaunch(exec, self.stream) };
        if s != CUDA_SUCCESS {
            unsafe { cuGraphExecDestroy(exec); cuMemPoolDestroy(graph_pool); }
            self.enabled = false;
            candle_core::bail!("First cuGraphLaunch failed: {s}");
        }
        unsafe { cudaStreamSynchronize(self.stream); }

        tracing::info!("CUDA graph: captured + launched for batch_size={batch_size}");

        let result = output.clone();
        self.graphs.insert(batch_size, CapturedGraph { exec, output, pool: graph_pool });
        Ok(result)
    }

    /// Cancel an in-progress capture and restore the original pool.
    pub fn cancel_capture(&self, graph_pool: CUmemoryPool, original_pool: CUmemoryPool) {
        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = cuStreamEndCapture(self.stream, &mut graph);
            if !graph.is_null() { cuGraphDestroy(graph); }
            cuDeviceSetMemPool(self.device_ordinal, original_pool);
            cuMemPoolDestroy(graph_pool);
        }
    }

    pub fn disable(&mut self) { self.enabled = false; }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        // CapturedGraph handles cleanup via its own Drop
    }
}
