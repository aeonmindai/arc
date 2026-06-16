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
    /// RUN-161: the deferred-free warmup pass has run. After the eager warmups
    /// (which only fill the alloc cache to peak-live), exactly one forward must
    /// run with the device in capture mode so the cache grows to the FULL
    /// per-forward allocation count (every alloc distinct, frees deferred).
    /// Generic: the caller toggles the device's capture mode; the runner just
    /// tracks that the pass is owed. See `try_take_deferred_pass`.
    deferred_pass_done: bool,
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
                stream,
                device_ordinal: 0,
                graphs: HashMap::new(),
                enabled: false,
                warmup_remaining: 0,
                deferred_pass_done: false,
            });
        }

        // Get device ordinal from the stream's context
        let ordinal = match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id as CUdevice,
            _ => 0,
        };

        tracing::info!("CUDA graph: non-null stream on device {ordinal}, capture enabled");

        Ok(Self {
            stream,
            device_ordinal: ordinal,
            graphs: HashMap::new(),
            enabled: true,
            warmup_remaining: warmup_steps,
            deferred_pass_done: false,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled && self.warmup_remaining == 0
    }

    /// Returns `true` exactly once, after warmup completes and before the graph
    /// is captured: the caller should then run ONE forward with the device's
    /// caching allocator in capture mode (`set_capture_mode(true)` ... forward
    /// ... `set_capture_mode(false)`) to grow the free pool to the full
    /// per-forward allocation count. Generic across models.
    pub fn try_take_deferred_pass(&mut self) -> bool {
        if self.enabled && self.warmup_remaining == 0 && !self.deferred_pass_done {
            self.deferred_pass_done = true;
            true
        } else {
            false
        }
    }

    pub fn tick_warmup(&mut self) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            if self.warmup_remaining == 0 {
                tracing::info!("CUDA graph: warmup done");
            }
            true
        } else {
            false
        }
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
            if s != CUDA_SUCCESS {
                candle_core::bail!("cuGraphLaunch failed: {s}");
            }
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
            unsafe {
                cuMemPoolDestroy(pool);
            }
            candle_core::bail!("cuMemPoolSetAttribute(RELEASE_THRESHOLD) failed: {s}");
        }

        Ok(pool)
    }

    /// Begin graph capture with a private memory pool.
    /// Returns the pool and saved original pool for restoration.
    pub fn begin_capture(
        &mut self,
        batch_size: usize,
    ) -> candle_core::Result<(CUmemoryPool, CUmemoryPool)> {
        // Create private pool
        let graph_pool = self.create_private_pool()?;

        // Save original pool
        let mut original_pool: CUmemoryPool = std::ptr::null_mut();
        let s = unsafe { cuDeviceGetMemPool(&mut original_pool, self.device_ordinal) };
        if s != CUDA_SUCCESS {
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            candle_core::bail!("cuDeviceGetMemPool failed: {s}");
        }

        // Install private pool
        let s = unsafe { cuDeviceSetMemPool(self.device_ordinal, graph_pool) };
        if s != CUDA_SUCCESS {
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
            candle_core::bail!("cuDeviceSetMemPool (install) failed: {s}");
        }

        // Sync stream before capture
        unsafe {
            cudaStreamSynchronize(self.stream);
        }

        // Begin capture
        // RELAXED tolerates cross-stream dependencies that candle's allocator
        // and helper streams create (THREAD_LOCAL rejects them with
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION). (RUN-161)
        let s = unsafe { cuStreamBeginCapture_v2(self.stream, CUstreamCaptureMode::RELAXED) };
        if s != CUDA_SUCCESS {
            // Restore original pool before bailing
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
            }
            unsafe {
                cuMemPoolDestroy(graph_pool);
            }
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
        // End capture. NOTE: the private pool stays installed as the device
        // default through instantiate + the first launch. Graph memory nodes
        // (candle's intermediate allocs recorded during capture) bind to the
        // pool active at instantiate, and the stream-ordered allocator backs
        // the capture-time virtual addresses at the first launch. Restoring the
        // original pool BEFORE that (the previous behavior) made the graph
        // allocate from the wrong pool -> the kernels' baked addresses were
        // never backed -> MMU read fault (Xid 31) on launch. Restore only AFTER
        // the first launch has materialized the graph's memory.
        let mut graph: CUgraph = std::ptr::null_mut();
        let s = unsafe { cuStreamEndCapture(self.stream, &mut graph) };
        if s != CUDA_SUCCESS {
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuStreamEndCapture failed: {s}");
        }

        // Instantiate (private pool still installed).
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let s = unsafe {
            cuGraphInstantiate_v2(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        unsafe {
            cuGraphDestroy(graph);
        }
        if s != CUDA_SUCCESS {
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("cuGraphInstantiate failed: {s}");
        }

        // First launch (private pool still installed -> graph memory is
        // allocated/backed here at the capture-time addresses).
        let s = unsafe { cuGraphLaunch(exec, self.stream) };
        if s != CUDA_SUCCESS {
            unsafe {
                cuDeviceSetMemPool(self.device_ordinal, original_pool);
                cuGraphExecDestroy(exec);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!("First cuGraphLaunch failed: {s}");
        }
        // Check the sync: an illegal access during graph EXECUTION surfaces
        // here (async), not at launch. Without this check it silently poisons
        // the CUDA context and the process dies later with no diagnostic.
        // cudaError: 700 = illegalAddress, 719 = launchFailure, 1 = invalidValue.
        let sync = unsafe { cudaStreamSynchronize(self.stream) };
        // Restore the original pool now that the graph's memory is materialized.
        // Subsequent replays reuse the already-allocated graph memory, so the
        // device default pool no longer needs to be the private one.
        unsafe {
            cuDeviceSetMemPool(self.device_ordinal, original_pool);
        }
        if sync != CUDA_SUCCESS {
            unsafe {
                cuGraphExecDestroy(exec);
                cuMemPoolDestroy(graph_pool);
            }
            self.enabled = false;
            candle_core::bail!(
                "Graph first-launch sync failed (async fault during graph execution): cudaError {sync}"
            );
        }

        tracing::info!("CUDA graph: captured + launched for batch_size={batch_size}");

        let result = output.clone();
        self.graphs.insert(
            batch_size,
            CapturedGraph {
                exec,
                output,
                pool: graph_pool,
            },
        );
        Ok(result)
    }

    /// Cancel an in-progress capture and restore the original pool.
    pub fn cancel_capture(&self, graph_pool: CUmemoryPool, original_pool: CUmemoryPool) {
        let mut graph: CUgraph = std::ptr::null_mut();
        unsafe {
            let _ = cuStreamEndCapture(self.stream, &mut graph);
            if !graph.is_null() {
                cuGraphDestroy(graph);
            }
            cuDeviceSetMemPool(self.device_ordinal, original_pool);
            cuMemPoolDestroy(graph_pool);
        }
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        // CapturedGraph handles cleanup via its own Drop
    }
}
